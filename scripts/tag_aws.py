"""
Tag AWS instances and EBS volumes

Tag definition standard can be found here:
    https://tmobileusa.sharepoint.com/:w:/r/sites/ \
          tpd_tt/cloudgovernance/_layouts/15/ \
          Doc.aspx?sourcedoc=%7B841927e1-306a-4463-9d24-5f461fb2326c%7D

environment is one of = MGT, STG, NPE, PRD
application = pcf
owner = cfops@t-mobile.com
"""
#pylint: disable=broad-except, bare-except, invalid-name, global-statement
import argparse
import itertools
import json
import logging
import os
import random
import sys
import time

import boto3
from botocore.exceptions import ClientError, WaiterError

OWNER = 'cfops@t-mobile.com'
VALID_ENVIRONMENTS = ['mgt', 'stg', 'npe', 'prd']
MAX_BACKOFF_TIME = 15.0
MAX_BACKOFF_JITTER = 1.0
PARAMS = {}
LOGGER = None
APPNAME = "tag_aws"

def get_param(param, default=None):
    """
    Return the value of the named parameter in command line or environment.
    If not set in either place then return the default.

    Note that this depends on the argparse default values being "None".

    :param param: string name of the requested parameter
    """
    return PARAMS.get(param) or os.environ.get(param.upper(), default)


class InstanceInfo():   #pylint: disable=too-few-public-methods
    """
    Encapsulate the information that instance/volume tagging needs
    to extract from the existing instance "Name" tag.
    """
    class InvalidEnvironment(Exception):
        """ Environment is not in the allowed list """
        pass

    class ErrorNoNameTag(Exception):
        """ No 'Name' tag found in tag list """
        pass

    def __init__(self, instance):
        """
        Initialize the tag information field(s).  If the tag cannot be
        initialized then this will raise an exception.  The caller should
        catch that and infer that the information is not available.

        Must set:
            environment (mgt)
            application (pcf)
            appname     (concourse_worker-w2)
            suffix      (49af159d-6a49-489f-ac91-88ae62748745)
        """
        LOGGER.debug("Looking for name tag")
        for tag in instance.tags:
            # Check every tag looking for 'Name'
            if tag['Key'].lower() == "name":
                fullname = tag['Value']
                break
        else:
            LOGGER.debug("No 'Name' tag found: %s", instance.tags)
            raise InstanceInfo.ErrorNoNameTag

        try:
            zoneinfo = instance.placement['AvailabilityZone'].split('-')
            self.zonesuffix = zoneinfo[-2][0] + zoneinfo[-1]
        except Exception as exn:
            LOGGER.debug("Error extracting zone suffix from %s: %s",
                         instance.placement['AvailabilityZone'], exn)
            raise exn

        LOGGER.debug("Parse instance info (fullname %s)", fullname)
        self._parse_instance_info(instance, fullname)
        LOGGER.debug("Parsing instance information succeeded")

    def _parse_instance_info(self, instance, fullname):
        """
        Gather tagging information from the instance.  We allow any
        failures here to raise uncaught so that the caller can skip
        tagging this instance.
        """
        LOGGER.debug("Parsing instance information for %s", fullname)
        keypair_name = instance.key_name
        (self.environment,
         self.application) = keypair_name.split('-')[0:2]
        (appname, self.suffix) = fullname.split('/')[0:2]
        self.appname = '-'.join([appname, self.zonesuffix])
        if self.environment not in VALID_ENVIRONMENTS:
            LOGGER.warning("Environment %s is not one of [%s]",
                           self.environment, ','.join(VALID_ENVIRONMENTS))
            raise InstanceInfo.InvalidEnvironment


def get_taggable_resources(ec2, instance_id=None):  #pylint: disable=too-many-locals
    """
    Given an ec2 object gather tag information on all running instances
    and all volumes attached to those instances.

    :param ec2: the ec2 object
    :return: dictionary information for instances, volumes, network interfaces
    """
    instances = {}
    volumes = {}
    interfaces = {}
    # for all instances, gather the information for tagging
    try:
        if instance_id:
            LOGGER.debug("Retrieve instance (id %s)", instance_id)
            ec2_instances = ec2.instances.filter(InstanceIds=[instance_id])
        else:
            states = ['running', 'pending']
            LOGGER.debug("Retrieve all instances in state: %s", ','.join(states))
            filters = [{'Name': 'instance-state-name', 'Values': states}]
            ec2_instances = ec2.instances.filter(Filters=filters)
    except Exception as exn:
        LOGGER.warning("Error fetching instance(s): %s", exn)
        raise exn

    for instance in ec2_instances:
        try:
            LOGGER.debug("Get instance tagging info for instance %s",
                         instance.id)
            tag_info = InstanceInfo(instance)
            LOGGER.debug("Created tag info for instance %s", instance.id)
        except:
            LOGGER.debug("Error getting tag info, skip tagging %s", instance.id)
            continue

        inst_descr = '-'.join([tag_info.environment,
                               tag_info.application,
                               tag_info.appname])
        if tag_info.suffix:
            inst_descr = '/'.join([inst_descr, tag_info.suffix])
        instances[instance.id] = {'owner': OWNER,
                                  'environment': tag_info.environment,
                                  'appname': tag_info.application,
                                  'description': inst_descr.lower(),
                                  'current_tags': instance.tags,
                                 }
        # for all the volumes, gather information for tagging
        LOGGER.debug("Create volume tagging info for instance %s", instance.id)
        for vol in instance.volumes.all():
            LOGGER.debug("Instance: %s, volume: %s, attachments: %d",
                         instance.id, vol.id, len(vol.attachments))
            for att in vol.attachments:
                device = att['Device']
                zonesplit = vol.availability_zone.split('-')
                zonestr = "{}{}".format(zonesplit[1][0], zonesplit[-1])
                vol_descr = '-'.join([tag_info.environment,
                                      tag_info.application,
                                      '-'.join(tag_info.appname.split('-')[:-1]),
                                      'volume',
                                      device[-1],
                                      zonestr])
                volumes[att['VolumeId']] = {'owner': OWNER,
                                            'environment': tag_info.environment,
                                            'appname': tag_info.application,
                                            'description': vol_descr.lower(),
                                            'current_tags': vol.tags,
                                           }
                LOGGER.debug("Created tagging info for " + \
                             "instance %s volume %s attachment %s",
                             instance.id, vol.id, device)

        # for all the instance interfaces, gather information for tagging
        LOGGER.debug("Create interface tagging info for instance %s", instance.id)
        for face in instance.network_interfaces:
            LOGGER.debug("Instance: %s, interface: %s",
                         instance.id, face.id)
            # 'face' object tag_set and availability_zone are not set.
            # Must make this call to retrieve those
            eni = ec2.NetworkInterface(face.id)
            eni_az = eni.availability_zone.split('-')
            eni_zsuff = eni_az[-2][0] + eni_az[-1]
            face_descr = '-'.join([tag_info.environment,
                                   tag_info.application,
                                   '-'.join(tag_info.appname.split('-')[:-1]),
                                   'network_interface',
                                   eni_zsuff])
            interfaces[face.id] = {'owner': OWNER,
                                   'environment': tag_info.environment,
                                   'appname': tag_info.application,
                                   'description': face_descr.lower(),
                                   'current_tags': eni.tag_set,
                                  }

    LOGGER.debug("Discovered %d instances, %d volumes to tag, %d interfaces",
                 len(instances), len(volumes), len(interfaces))
    return instances, volumes, interfaces


def apply_tags(ec2, resource_id, tag_data):
    """
    Call "create_tags".  If rate limit is exceeded then back off and try again.

    Note(s):
    1. this will only exit if some uncaught exception occurs or the
       create_tags() succeeds.  The expectation is that with increasing
       backoff the create will eventually succeed.
    2. this function will not return until one of the above mentioned
       conditions is met.  In other words this is a blocking call.
    3. subsequent calls here will not do an initial wait and will reset
       the delay value, so the blocking time will not increase from one
       call to the next (unless AWS causes it to).

    :param ec2: ec2 object
    :param resource_id: id of resource to be tagged
    :param tag_data: list of tag key/values
    :return:
    """
    delay = 0.25
    jitter = 0.0
    for attempt in itertools.count(1):
        try:
            if not (get_param('deploy') or get_param('print')):
                log_to = LOGGER.info
            else:
                log_to = LOGGER.debug
            log_to("ec2.create_tags (Resources=[%s], Tags=%s)",
                   resource_id, json.dumps(tag_data, sort_keys=True))
            if get_param('deploy'):
                ec2.create_tags(Resources=[resource_id], Tags=tag_data)
                LOGGER.debug("Create tags succeeded (resource %s: %d attempts)",
                             resource_id, attempt)
            break
        except (ClientError, WaiterError) as exn:
            # RateLimit or Throttling error
            LOGGER.debug("Handling error; %s", exn)
            if isinstance(exn, ClientError):
                err_code = exn.response['Error']['Code']
            elif isinstance(exn, WaiterError):
                err_code = exn.last_response['Error']['Code']
            else:
                LOGGER.warning("Re-raising Client/Waiter exception while tagging: %s",
                               exn)
                raise
            if err_code in ['Throttling', 'RequestLimitExceeded']:
                LOGGER.info("Limit error on id %s: %s, retry in %d seconds",
                            resource_id, err_code, delay + jitter)
                time.sleep(delay + jitter)
                max_backoff = float(get_param('max_backoff', MAX_BACKOFF_TIME))
                max_jitter = float(get_param('max_jitter', MAX_BACKOFF_JITTER))
                delay = min(max_backoff, delay * 2)
                jitter = round(random.uniform(0, max_jitter), 2)
            else:
                LOGGER.debug("Re-raising %s error while tagging: %s",
                             err_code, exn)
                raise
        except Exception as exn:
            LOGGER.debug("Re-raising uncaught exception while tagging: %s",
                         exn)
            raise
    LOGGER.debug("Exit apply_tag: resource %s, delay/jitter %d/%d, attempts %d",
                 resource_id, delay, jitter, attempt)


def apply_instance_tags(ec2, instance_tags):
    """
    Given a dict of resource_id as key and tag info dict as value, form
    the instance tags and apply them

    :param ec2: the ec2 object
    :param instance_tags: dict of instance data from get_taggable_resources()
    :return: number of instances tagged
    """
    LOGGER.debug("Apply %d instance tags", len(instance_tags))
    tag_map = {'Name': 'description',
               'Application': 'appname',
               'Environment': 'environment',
               'Owner': 'owner'}
    return map_and_apply_tags(ec2, instance_tags, tag_map)


def apply_volume_tags(ec2, volume_tags):
    """
    Given a dict of resource_id as key and tag info dict as value, form
    the volume tag(s) and apply them

    :param ec2: the ec2 object
    :param volume_tags: dict of volume data from get_taggable_resources()
    :return: number of volumes tagged
    """
    LOGGER.debug("Apply %d volume tags", len(volume_tags))
    tag_map = {'Name': 'description',
               'Application': 'appname',
               'Environment': 'environment',
               'Owner': 'owner'
              }
    return map_and_apply_tags(ec2, volume_tags, tag_map)


def apply_interface_tags(ec2, interface_tags):
    """
    Given a dict of resource_id as key and tag info dict as value, form
    the interface tags and apply them

    :param ec2: the ec2 object
    :param instance_tags: dict of interface data from get_taggable_resources()
    :return: number of interfaces tagged
    """
    LOGGER.debug("Apply %d interface tags", len(interface_tags))
    tag_map = {'Name': 'description',
               'Application': 'appname',
               'Environment': 'environment',
               'Owner': 'owner'}
    return map_and_apply_tags(ec2, interface_tags, tag_map)


def map_and_apply_tags(ec2, resource_tags, tag_map):
    """
    Given a dict of resource_id as key and tag info dict as value, form
    the resource tag(s) and apply them

    :param ec2: the ec2 object
    :param resource_tags: dict of resource data from get_taggable_resources()
    :return: number of resources tagged
    """
    def _dict_from_list(taglist):
        """
        Convert AWS tag list to a flat dictionary.
        AWS tag lists are [{Key: keyname1, Value: val1},
                           {Key: keyname2, Value: val2},
                           ...(and so on)]
        This assumes that (for our purposes) the keynames are unique.
        """
        return {tag['Key']: tag['Value'] for tag in taglist if tag} \
                if taglist else {}

    def _tags_is_subset(new_tags, current_tags):
        """
        Check if new_tags is a subset of current_tags.
        """
        curr = _dict_from_list(current_tags)
        new = _dict_from_list(new_tags)
        return True if new.items() <= curr.items() else False

    def make_tags(info):
        """
        Create a list of dicts suitable for passing to boto3 create_tags()

        :param info: dictionary to convert to tag format
        :return: list of dicts
        """
        return [{'Key': k, 'Value': v} for k, v in info.items()]

    LOGGER.debug("Apply tags to %d resources", len(resource_tags))
    tag_count = 0
    for resource_id, data in resource_tags.items():
        tags = {k: data[v] for k, v in tag_map.items()}
        tags = make_tags(tags)
        if not _tags_is_subset(tags, data.get('current_tags')):
            apply_tags(ec2, resource_id, tags)
            tag_count += 1
        else:
            LOGGER.info("Tag on resource %s is current, no re-tag", resource_id)
            LOGGER.debug("Resource %s current tag [%s] subset of [%s]",
                         resource_id, tags, data.get('current_tags', {}))
    return tag_count


def setup_logger():
    """
    Initialize the (global) logger instance.

    Note that bosh uses AWS REST calls and intercepts logging output,
    so if no "name" is set in the call to getLogger, then logged output
    includes all the bosh/AWS log messages.  We get around this by
    specifying a name to getLogger.  If "verbose" mode is set then we
    allow bosh to log to the same stream as us, otherwise we log to our
    own stream.
    """
    global LOGGER
    name = None if get_param('verbose') else APPNAME
    LOGGER = logging.getLogger(name)
    LOGGER.addHandler(logging.StreamHandler(sys.stdout))
    LOGGER.setLevel(get_param('log_level', 'INFO').upper())

    boto_loglevel = get_param('boto_logging', "").upper()
    if boto_loglevel:
        LOGGER.info("setting boto3 log level %s", boto_loglevel)
        boto3.set_stream_logger(level=boto_loglevel)


def tag_handler(event=None, context=None):  #pylint: disable=too-many-locals, unused-argument
    """
    Main lambda function entry point
    """
    # Get the triggering instance or all EC2 running instances if none
    setup_logger()
    LOGGER.debug("Received event: %s", event)
    try:
        instance_id = get_param('instance_id') or event['detail']['instance-id']
    except (KeyError, TypeError):
        instance_id = None
    LOGGER.info("Tagging instance id %s", instance_id or "(all)")
    ec2 = boto3.resource('ec2', region_name='us-west-2')
    # Get all volumes and instances to be tagged, and the information to
    # be used in the tagging
    try:
        (instances,
         volumes,
         interfaces) = get_taggable_resources(ec2, instance_id=instance_id)
    except ClientError as exn:
        err_code = exn.response['Error']['Code']
        if err_code.startswith('InvalidInstanceID'):
            LOGGER.info("Error: invalid instance id: %s", instance_id)
        raise exn
    # Write the tags
    inst_tagged = apply_instance_tags(ec2, instances)
    vols_tagged = apply_volume_tags(ec2, volumes)
    ifaces_tagged = apply_interface_tags(ec2, interfaces)

    if get_param('print'):
        resources = [instances, volumes, interfaces]
        for resource in resources:
            ids = sorted([i for i in resource])
            for rid in ids:
                print("[{}] {}".format(rid, resource[rid]['description']))

    LOGGER.info("Tagged/taggable: instances %d/%d, volumes %d/%d, interfaces %d/%d",
                inst_tagged, len(instances),
                vols_tagged, len(volumes),
                ifaces_tagged, len(interfaces)
               )
    return {'number_taggable_volumes': len(volumes),
            'number_taggable_instances': len(instances),
            'number_taggable_interfaces': len(interfaces),
            'number_volumes_tagged': vols_tagged,
            'number_instances_tagged': inst_tagged,
            'number_interfaces_tagged': ifaces_tagged,
           }


if __name__ == "__main__":
    # Entry for debugging purposes.  Lambda will enter the handler function directly
    parser = argparse.ArgumentParser()
    # See note in get_param(): default values should be "None"
    parser.add_argument("-b", "--max-backoff",
                        default=None, type=float,
                        help="maximum AWS backoff time seconds (float)")
    parser.add_argument("-d", "--deploy", action="store_true",
                        default=False,
                        help="deploy: execute the create_tags() commands (default False)")
    parser.add_argument("-i", "--instance-id",
                        default=None, type=str,
                        help="Specific EC2 instance (simulates lambda event)")
    parser.add_argument("-j", "--max-jitter",
                        default=None, type=float,
                        help="Maximum AWS backoff jitter seconds (float)")
    parser.add_argument("-l", "--log-level",
                        default=None, type=str,
                        help="logging level (info, debug, ...)")
    parser.add_argument("-t", "--boto-logging",
                        default=None, type=str,
                        help="set boto3 stream logging (info, debug, ...)")
    parser.add_argument("-p", "--print", action="store_true",
                        default=False,
                        help="print volume and instance id/names before exit")
    parser.add_argument("-v", "--verbose", action="store_true",
                        default=None,
                        help="turn on verbose mode (include AWS calls)")
    args = parser.parse_args()
    PARAMS = vars(args)
    if PARAMS.get('verbose'):
        PARAMS['log_level'] = 'DEBUG'

    if get_param('deploy'):
        decide = input("Not dry-run: tags will be created.  Are you sure? Y[es]: ")
        if decide.lower() not in ['y', 'yes']:
            print("OK ... abort")
            exit(0)
    else:
        print("Dry run only, no create_tags() calls issued")

    tag_handler()
